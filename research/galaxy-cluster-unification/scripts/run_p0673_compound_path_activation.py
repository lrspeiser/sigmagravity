#!/usr/bin/env python3
"""Audit frozen compound-path activation on registered and spent maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import manifest_sha256, sha256
from run_p0668_registered_multipole_3d_activation import common_surface

from voidscreen.compound_activation_3d import exact_compound_path_activation_3d
from voidscreen.metric_lensing_3d import (
    KPC_M,
    lift_surface_density_msun_kpc2_to_si_volume,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0673_compound_path_activation.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted(values, density):
    return float(np.sum(np.asarray(values) * density) / np.sum(density))


def activation_kwargs(protocol):
    candidate = protocol["candidate"]
    return {
        "a0": float(candidate["a0_m_s2"]),
        "coherence_length": float(candidate["coherence_length_kpc"]) * KPC_M,
        "coherence_power": float(candidate["coherence_power"]),
    }


def score_volume(case, domain, scenario, stars, gas, spacing_m, protocol):
    result = exact_compound_path_activation_3d(
        stars,
        gas,
        spacing_m,
        **activation_kwargs(protocol),
    )
    total = stars + gas
    return {
        "case": case,
        "domain": domain,
        "scenario": scenario,
        "mass_weighted_sigma": weighted(result.sigma, total),
        "mass_weighted_elementary_probability": weighted(
            result.elementary_probability,
            total,
        ),
        "mass_weighted_coherent_opportunities": weighted(
            result.coherent_opportunities,
            total,
        ),
        "mass_weighted_trace_length_kpc": weighted(
            result.local.trace_length,
            total,
        )
        / KPC_M,
        "multipole_power_gate": result.multipole.gate,
        "multipole_amplitude_gate": result.amplitude_gate,
        "sigma_minimum": float(np.min(result.sigma)),
        "sigma_maximum": float(np.max(result.sigma)),
        "minimum_constitutive_eigenvalue_proxy": float(
            np.min(result.minimum_eigenvalue_proxy)
        ),
        "all_coefficients_finite": bool(
            np.all(np.isfinite(result.sigma))
            and np.all(np.isfinite(result.minimum_eigenvalue_proxy))
        ),
    }


def score_surface(case, domain, scenario, stars, gas, axis, protocol):
    cells = int(protocol["map_inputs"]["common_grid_cells"])
    star_surface = common_surface(stars, cells)
    gas_surface = common_surface(gas, cells)
    cell_kpc = float((axis[-1] - axis[0]) / (cells - 1))
    z_kpc = np.linspace(float(axis[0]), float(axis[-1]), cells)
    star_volume, _ = lift_surface_density_msun_kpc2_to_si_volume(
        star_surface,
        z_kpc,
        cell_kpc=cell_kpc,
    )
    gas_volume, _ = lift_surface_density_msun_kpc2_to_si_volume(
        gas_surface,
        z_kpc,
        cell_kpc=cell_kpc,
    )
    return score_volume(
        case,
        domain,
        scenario,
        star_volume,
        gas_volume,
        cell_kpc * KPC_M,
        protocol,
    )


def registered_scores(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    paths = []
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        for scenario, scale in inputs["galaxy_stellar_scale_sensitivity"].items():
            rows.append(
                score_surface(
                    path.stem,
                    "registered_galaxy_baryons_only",
                    scenario,
                    stars * float(scale),
                    gas,
                    axis,
                    protocol,
                )
            )
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            for scenario, keys in inputs["cluster_sensitivity_maps"].items():
                rows.append(
                    score_surface(
                        path.stem.replace("_baryons", ""),
                        "registered_cluster_baryons_only",
                        scenario,
                        data[keys[0]].astype(float),
                        data[keys[1]].astype(float),
                        axis,
                        protocol,
                    )
                )
    return pd.DataFrame(rows), paths


def radial_null(protocol):
    axis = np.linspace(-8.0, 8.0, 25)
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

    def gaussian(width, mass):
        values = np.exp(-0.5 * (x * x + y * y + z * z) / width**2)
        return values * mass / (np.sum(values) * spacing**3)

    stars = gaussian(0.8, 0.3)
    gas = gaussian(1.6, 0.7)
    result = exact_compound_path_activation_3d(
        stars,
        gas,
        spacing,
        gravitational_constant=1.0,
        a0=0.1,
        coherence_length=2.0,
        coherence_power=float(protocol["candidate"]["coherence_power"]),
    )
    return weighted(result.sigma, stars + gas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0673_coefficient_score":
        raise RuntimeError("P0673 protocol is not frozen")
    development = read_json(ROOT / protocol["development_parent"])
    registered_parent = read_json(ROOT / protocol["registered_parent"])
    spent_parent = read_json(ROOT / protocol["spent_map_parent"])
    topology = development["topology"]["tensor_absolute_P0669"]
    failed_single_root = bool(
        development["status"] == "fail"
        and topology["missing_multiplicity_families"] == 7
        and topology["critical_curve_present_families"] == 0
    )
    scores, map_paths = registered_scores(protocol)
    spent_path = ROOT / protocol["map_inputs"]["spent_RXJ2129_cube"]
    with np.load(spent_path) as data:
        axis = data["axis_kpc"].astype(float)
        spent = score_volume(
            "RXJ2129",
            "spent_cluster_baryons_only",
            "nominal",
            data["stellar_volume_density_kg_m3"].astype(float),
            data["gas_volume_density_kg_m3"].astype(float),
            float(axis[1] - axis[0]) * KPC_M,
            protocol,
        )
    summary = (
        scores.groupby(["domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_sigma=("mass_weighted_sigma", "median"),
            minimum_sigma=("mass_weighted_sigma", "min"),
            maximum_sigma=("mass_weighted_sigma", "max"),
            median_opportunities=("mass_weighted_coherent_opportunities", "median"),
        )
    )
    nominal = summary[summary.scenario.eq("nominal")].set_index("domain")
    galaxy_domain = "registered_galaxy_baryons_only"
    cluster_domain = "registered_cluster_baryons_only"
    galaxy_sigma = float(nominal.loc[galaxy_domain, "median_sigma"])
    cluster_sigma = float(nominal.loc[cluster_domain, "median_sigma"])
    ratio = cluster_sigma / max(galaxy_sigma, np.finfo(float).tiny)
    sensitivity = {}
    for scenario in protocol["map_inputs"]["galaxy_stellar_scale_sensitivity"]:
        block = summary[summary.scenario.eq(scenario)].set_index("domain")
        sensitivity[scenario] = float(
            block.loc[cluster_domain, "median_sigma"]
            / max(float(block.loc[galaxy_domain, "median_sigma"]), np.finfo(float).tiny)
        )
    radial_sigma = radial_null(protocol)
    galaxy_count = int(scores[scores.domain.eq(galaxy_domain)].case.nunique())
    cluster_count = int(scores[scores.domain.eq(cluster_domain)].case.nunique())
    maximum_sigma = max(float(scores.sigma_maximum.max()), float(spent["sigma_maximum"]))
    minimum_eigenvalue = min(
        float(scores.minimum_constitutive_eigenvalue_proxy.min()),
        float(spent["minimum_constitutive_eigenvalue_proxy"]),
    )
    bounded = bool(
        scores.all_coefficients_finite.all()
        and spent["all_coefficients_finite"]
        and float(scores.sigma_minimum.min()) >= 0.0
        and float(spent["sigma_minimum"]) >= 0.0
        and maximum_sigma < 1.0
    )
    gates = protocol["predeclared_progression_gates"]
    candidate = protocol["candidate"]
    gate_results = {
        "P0672_failure_mode": failed_single_root
        is bool(gates["P0672_failed_single_root_topology"]),
        "P0669_registered_parent": bool(registered_parent["all_progression_gates_pass"])
        is bool(gates["P0669_registered_parent_pass"]),
        "P0670_spent_parent": bool(spent_parent["all_progression_gates_pass"])
        is bool(gates["P0670_spent_map_parent_pass"]),
        "galaxy_coverage": galaxy_count == int(gates["registered_galaxy_count"]),
        "cluster_coverage": cluster_count == int(gates["registered_cluster_count"]),
        "radial_null": radial_sigma
        <= gates["radial_cocentered_synthetic_mass_weighted_sigma_max"],
        "galaxy_channel_small": galaxy_sigma
        <= gates["registered_galaxy_nominal_median_sigma_max"],
        "cluster_channel_nonperturbative": cluster_sigma
        >= gates["registered_cluster_nominal_median_sigma_min"],
        "nominal_domain_separation": ratio
        >= gates["registered_cluster_to_galaxy_nominal_median_sigma_ratio_min"],
        "mass_sensitivity": min(sensitivity.values())
        >= gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"],
        "spent_cluster_channel": float(spent["mass_weighted_sigma"])
        >= gates["spent_RXJ2129_mass_weighted_sigma_min"],
        "bounded_sigma": bounded
        is bool(gates["sigma_finite_and_in_half_open_unit_interval"]),
        "sigma_cap": maximum_sigma <= gates["sigma_global_maximum_max"],
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_proxy_strictly_positive"]),
        "no_new_constants": int(candidate["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(candidate["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "no_new_raw_lens_score": not bool(gates["new_raw_lens_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "radial_cocentered_synthetic_mass_weighted_sigma": radial_sigma,
        "registered_galaxy_nominal_median_sigma": galaxy_sigma,
        "registered_cluster_nominal_median_sigma": cluster_sigma,
        "registered_cluster_to_galaxy_nominal_median_sigma_ratio": ratio,
        "mass_sensitivity_cluster_to_galaxy_sigma_ratios": sensitivity,
        "spent_RXJ2129_mass_weighted_sigma": spent["mass_weighted_sigma"],
        "spent_RXJ2129_mass_weighted_elementary_probability": spent[
            "mass_weighted_elementary_probability"
        ],
        "spent_RXJ2129_mass_weighted_coherent_opportunities": spent[
            "mass_weighted_coherent_opportunities"
        ],
        "maximum_sigma": maximum_sigma,
        "minimum_constitutive_eigenvalue_proxy": minimum_eigenvalue,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "registered_compound_scores.csv", index=False)
    summary.to_csv(output / "domain_summary.csv", index=False)
    (output / "spent_RXJ2129_score.json").write_text(
        json.dumps(spent, indent=2),
        encoding="utf-8",
    )
    report = {
        "report_version": "P0673-COMPOUND-PATH-ACTIVATION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_new_spent_field_solve": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(ROOT / "src/voidscreen/compound_activation_3d.py"),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "spent_map_sha256": sha256(spent_path),
        "metrics": metrics,
        "gate_results": gate_results,
        "new_raw_lens_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    nominal_scores = scores[scores.scenario.eq("nominal")].sort_values(
        ["domain", "mass_weighted_sigma"]
    )
    colors = nominal_scores.domain.str.contains("cluster").map(
        {False: "#3274a1", True: "#d95f02"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(nominal_scores.case, nominal_scores.mass_weighted_sigma, color=colors)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set(title="Compound activation", ylabel="mass-weighted sigma")
    axes[1].bar(
        nominal_scores.case,
        nominal_scores.mass_weighted_coherent_opportunities,
        color=colors,
    )
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1].set(title="Measured coherent opportunities", ylabel="mass-weighted N")
    figure.tight_layout()
    figure.savefig(output / "p0673_compound_path_activation.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    text = f"""# P0673 compound-path activation

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Registered galaxy/cluster nominal median sigma: **{galaxy_sigma:.6g} / {cluster_sigma:.6g}**.
- Nominal cluster/galaxy ratio: **{ratio:.4g}x**.
- Weakest mass-sensitivity ratio: **{min(sensitivity.values()):.4g}x**.
- Spent RX J2129 mass-weighted sigma: **{spent['mass_weighted_sigma']:.6g}**.
- Global maximum sigma / minimum eigenvalue proxy: **{maximum_sigma:.6g} / {minimum_eigenvalue:.6g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- New raw-lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
