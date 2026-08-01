#!/usr/bin/env python3
"""Add observed CLASH BCG stellar densities to the ACCEPT bridge test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_cpr0_accept_clash_bridge import (
    MU_E,
    PROTON_G,
    build_clash_sample,
    domain_metrics,
    load_clash_accelerations,
    run_cv_suite,
    score_fixed_transfers,
    sha256,
)
from scripts.run_cpr0_joint_bcg_lensing import prepare_domain
from scripts.run_cpr0_manga_bcg_coherence import load_sample as load_bcg_sample
from voidscreen.accept_profiles import (
    interpolate_electron_density_cm3,
    load_accept_profiles,
)
from voidscreen.host_profiles import hernquist_local_density_from_total_mass

TABLE1_COLUMNS = (
    "name",
    "redshift",
    "coordinates",
    "band",
    "sersic_n",
    "effective_radius_kpc",
    "effective_radius_error_kpc",
    "central_radius_kpc",
    "stellar_mass_1e11_msun",
    "gas_mass_1e11_msun",
    "gas_mass_error_1e11_msun",
    "total_mass_1e11_msun",
    "total_mass_error_1e11_msun",
    "cluster",
)


def load_clash_bcg_properties(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\|",
        engine="python",
        header=None,
        names=TABLE1_COLUMNS,
        skipinitialspace=True,
    )
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["cluster"] = frame["cluster"].astype(str).str.strip()
    required = [
        "redshift",
        "sersic_n",
        "effective_radius_kpc",
        "central_radius_kpc",
        "stellar_mass_1e11_msun",
    ]
    frame[required] = frame[required].apply(pd.to_numeric, errors="raise")
    if (
        len(frame) != 20
        or frame["cluster"].nunique() != 20
        or np.any(frame[required].to_numpy(dtype=float) <= 0.0)
    ):
        raise ValueError("CLASH BCG table does not contain 20 valid systems")
    return frame


def build_stellar_augmented_sample(
    accept_path: Path,
    clash_path: Path,
    table1_path: Path,
    name_map: dict[str, str],
    *,
    gas_density_scale: float = 1.0,
    stellar_mass_scale: float = 1.0,
) -> pd.DataFrame:
    if gas_density_scale <= 0.0 or stellar_mass_scale <= 0.0:
        raise ValueError("density and stellar-mass scales must be positive")
    properties = load_clash_bcg_properties(table1_path).set_index("cluster")
    outer, _ = build_clash_sample(
        accept_path,
        clash_path,
        name_map,
        minimum_radius_kpc=100.0,
        density_scale=gas_density_scale,
    )
    outer = outer.copy()
    outer["gas_density_g_cm3"] = outer["local_density_g_cm3"]
    outer["stellar_density_g_cm3"] = [
        float(
            hernquist_local_density_from_total_mass(
                stellar_mass_scale
                * float(properties.loc[row.system, "stellar_mass_1e11_msun"])
                * 1.0e11,
                row.radius_kpc,
                float(properties.loc[row.system, "effective_radius_kpc"]),
            )
        )
        for row in outer.itertuples()
    ]
    outer["local_density_g_cm3"] = (
        outer["gas_density_g_cm3"] + outer["stellar_density_g_cm3"]
    )
    outer["density_source"] = "ACCEPT gas + Hernquist BCG"

    profiles = load_accept_profiles(accept_path)
    available = set(profiles["name"])
    clash = load_clash_accelerations(clash_path)
    central = clash[clash["radius_kpc"] < 100.0]
    central_rows = []
    for row in central.itertuples():
        cluster = str(row.cluster)
        prop = properties.loc[cluster]
        stellar_density = float(
            hernquist_local_density_from_total_mass(
                stellar_mass_scale * float(prop["stellar_mass_1e11_msun"]) * 1.0e11,
                float(row.radius_kpc),
                float(prop["effective_radius_kpc"]),
            )
        )
        gas_density = 0.0
        accept_name = name_map.get(cluster, "")
        if accept_name in available:
            profile = profiles[profiles["name"] == accept_name]
            measured_min = float(profile["radius_kpc"].min())
            measured_max = float(profile["radius_kpc"].max())
            if measured_min <= float(row.radius_kpc) <= measured_max:
                nelec = float(
                    interpolate_electron_density_cm3(
                        profile, [float(row.radius_kpc)]
                    )[0]
                )
                gas_density = gas_density_scale * MU_E * PROTON_G * nelec
        central_rows.append(
            {
                "domain": "cluster",
                "system": cluster,
                "accept_name": accept_name,
                "radius_kpc": float(row.radius_kpc),
                "log_gbar": float(row.log_gbar),
                "log_gobs": float(row.log_gobs),
                "err_log_gbar": float(row.err_log_gbar),
                "err_log_gobs": float(row.err_log_gobs),
                "electron_density_cm3": (
                    gas_density / (gas_density_scale * MU_E * PROTON_G)
                    if gas_density > 0.0
                    else np.nan
                ),
                "gas_density_g_cm3": gas_density,
                "stellar_density_g_cm3": stellar_density,
                "local_density_g_cm3": gas_density + stellar_density,
                "coherence": 0.0,
                "density_source": (
                    "ACCEPT gas + Hernquist BCG"
                    if gas_density > 0.0
                    else "Hernquist BCG; gas below ACCEPT coverage"
                ),
            }
        )
    sample = pd.concat([pd.DataFrame(central_rows), outer], ignore_index=True, sort=False)
    return sample.sort_values(["system", "radius_kpc"]).reset_index(drop=True)


def subset_metrics(
    sample: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray
) -> dict:
    return domain_metrics(
        sample.loc[mask].reset_index(drop=True), np.asarray(prediction)[mask]
    )["cluster"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "cpr0_accept_clash_bcg_stellar_protocol.json",
    )
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=ROOT / "configs" / "cpr0_accept_clash_bridge_protocol.json",
    )
    parser.add_argument(
        "--bcg-protocol",
        type=Path,
        default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json",
    )
    parser.add_argument(
        "--accept",
        type=Path,
        default=ROOT / "data" / "raw" / "accept_cavagnolo2009" / "all_profiles.dat.txt",
    )
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--table1",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "table1.dat",
    )
    parser.add_argument(
        "--tian",
        type=Path,
        default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv",
    )
    parser.add_argument(
        "--dynpop",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits",
    )
    parser.add_argument(
        "--gas-only-report",
        type=Path,
        default=ROOT / "results" / "cpr0_accept_clash_bridge" / "report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "cpr0_accept_clash_bcg_stellar",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    bcg_protocol = json.loads(args.bcg_protocol.read_text(encoding="utf-8"))
    sample = build_stellar_augmented_sample(
        args.accept, args.clash, args.table1, base["cluster_name_map"]
    )
    bcg = prepare_domain(
        load_bcg_sample(args.tian, args.dynpop, bcg_protocol), "BCG"
    )
    joint = pd.concat([bcg, sample], ignore_index=True, sort=False)
    cv = protocol["cross_validation"]
    amplitude = float(cv["fixed_sigma_response_amplitude"])
    cluster_metrics, cluster_fits, cluster_predictions = run_cv_suite(sample, cv)
    joint_metrics, joint_fits, joint_predictions = run_cv_suite(joint, cv)
    fixed_metrics, fixed_predictions = score_fixed_transfers(
        sample, base["fixed_parameters"], amplitude
    )

    inner_mask = sample["radius_kpc"].to_numpy(dtype=float) < 100.0
    outer_mask = ~inner_mask
    radial_breakdown = {}
    for label, prediction in cluster_predictions.items():
        radial_breakdown[label] = {
            "central": subset_metrics(sample, prediction, inner_mask),
            "outer": subset_metrics(sample, prediction, outer_mask),
        }

    sensitivity = {}
    for gas_scale in protocol["observable_construction"][
        "gas_density_scale_sensitivity"
    ]:
        for stellar_scale in protocol["observable_construction"][
            "stellar_mass_scale_sensitivity"
        ]:
            varied = build_stellar_augmented_sample(
                args.accept,
                args.clash,
                args.table1,
                base["cluster_name_map"],
                gas_density_scale=float(gas_scale),
                stellar_mass_scale=float(stellar_scale),
            )
            varied_metrics, _ = score_fixed_transfers(
                varied, base["fixed_parameters"], amplitude
            )
            sensitivity[
                f"gas_{float(gas_scale):.1f}_stellar_{float(stellar_scale):.1f}"
            ] = varied_metrics

    gas_only = json.loads(args.gas_only_report.read_text(encoding="utf-8"))
    gas_only_rg = gas_only["all_measured_radii_sensitivity"]["metrics"]["RG_CV"]
    gates = protocol["advance_gates"]
    cluster_rg = cluster_metrics["RG_CV"]["cluster"]
    cluster_constant = cluster_metrics["constant_epsilon_CV"]["cluster"]
    joint_rg = joint_metrics["RG_CV"]
    joint_cpr0 = joint_metrics["CPR0_CV"]
    rhoc = [
        row["parameters"]["log10_rho_c_g_cm3"]
        for row in joint_fits["RG_CV"]
    ]
    gate_audit = {
        "cluster_systems": sample["system"].nunique() >= gates["cluster_systems_min"],
        "cluster_points": len(sample) >= gates["cluster_points_min"],
        "locked_prior_RG_cluster_RMSE": fixed_metrics["locked_prior_joint_RG"][
            "cluster"
        ]["equal_system_RMSE_dex"]
        <= gates["locked_prior_RG_cluster_RMSE_dex_max"],
        "cluster_RG_CV_RMSE": cluster_rg["equal_system_RMSE_dex"]
        <= gates["cluster_RG_CV_RMSE_dex_max"],
        "cluster_RG_CV_improves_constant": cluster_constant[
            "equal_system_RMSE_dex"
        ]
        - cluster_rg["equal_system_RMSE_dex"]
        >= gates["cluster_RG_CV_improvement_vs_constant_min_dex"],
        "central_points_RG_CV_RMSE": radial_breakdown["RG_CV"]["central"][
            "equal_system_RMSE_dex"
        ]
        <= gates["central_points_RG_CV_RMSE_dex_max"],
        "joint_RG_BCG_RMSE": joint_rg["BCG"]["equal_system_RMSE_dex"]
        <= gates["joint_RG_BCG_RMSE_dex_max"],
        "joint_RG_cluster_RMSE": joint_rg["cluster"]["equal_system_RMSE_dex"]
        <= gates["joint_RG_cluster_RMSE_dex_max"],
        "joint_RG_equal_domain_RMSE": joint_rg["equal_domain_RMSE_dex"]
        <= gates["joint_RG_equal_domain_RMSE_dex_max"],
        "CPR0_improves_RG": joint_rg["equal_domain_RMSE_dex"]
        - joint_cpr0["equal_domain_RMSE_dex"]
        >= gates["CPR0_improvement_vs_density_only_RG_min_dex"],
        "cluster_radial_residual_slope": abs(
            cluster_rg["radial_residual_slope_dex_per_dex"]
        )
        <= gates["cluster_absolute_radial_residual_slope_dex_per_dex_max"],
        "cluster_residual_density_correlation": abs(
            cluster_rg["residual_log_density_correlation"]
        )
        <= gates["cluster_absolute_residual_density_correlation_max"],
        "rho_c_fold_range": max(rhoc) - min(rhoc)
        <= gates["log10_rho_c_fold_range_max_dex"],
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for selection, frame, collection in (
        ("cluster", sample, {**cluster_predictions, **fixed_predictions}),
        ("joint", joint, joint_predictions),
    ):
        for label, prediction in collection.items():
            block = frame.copy()
            block["selection"] = selection
            block["model"] = label
            block["predicted_log_gobs"] = prediction
            block["residual_dex"] = prediction - block["log_gobs"].to_numpy(dtype=float)
            prediction_rows.append(block)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    sample.to_csv(args.output / "matched_density_sample.csv", index=False)

    inner = sample[inner_mask]
    outer = sample[outer_mask]
    report = {
        "status": "completed ACCEPT gas plus observed CLASH BCG local-density test",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "base_protocol_sha256": sha256(args.base_protocol),
            "bcg_protocol_sha256": sha256(args.bcg_protocol),
            "accept_sha256": sha256(args.accept),
            "clash_sha256": sha256(args.clash),
            "table1_sha256": sha256(args.table1),
            "tian_bcg_sha256": sha256(args.tian),
            "dynpop_sha256": sha256(args.dynpop),
            "gas_only_report_sha256": sha256(args.gas_only_report),
        },
        "sample": {
            "cluster_systems": int(sample["system"].nunique()),
            "cluster_points": len(sample),
            "central_points": int(inner_mask.sum()),
            "central_points_with_ACCEPT_gas": int(
                (inner["gas_density_g_cm3"] > 0.0).sum()
            ),
            "outer_points": int(outer_mask.sum()),
            "log10_total_density_range": [
                float(np.log10(sample["local_density_g_cm3"]).min()),
                float(np.log10(sample["local_density_g_cm3"]).max()),
            ],
            "median_central_stellar_density_fraction": float(
                np.median(inner["stellar_density_g_cm3"] / inner["local_density_g_cm3"])
            ),
            "median_outer_stellar_density_fraction": float(
                np.median(outer["stellar_density_g_cm3"] / outer["local_density_g_cm3"])
            ),
        },
        "cluster_only_metrics": cluster_metrics,
        "cluster_only_fold_fits": cluster_fits,
        "central_outer_breakdown": radial_breakdown,
        "fixed_transfer_metrics": fixed_metrics,
        "joint_BCG_plus_CLUSTER": {
            "metrics": joint_metrics,
            "fold_fits": joint_fits,
        },
        "comparison_to_gas_only_all_radii": {
            "gas_only_RG_CV": gas_only_rg,
            "gas_plus_BCG_RG_CV": cluster_metrics["RG_CV"],
        },
        "fixed_transfer_density_scale_sensitivity": sensitivity,
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
