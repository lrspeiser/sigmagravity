#!/usr/bin/env python3
"""Fit and globally audit the frozen compound RX J2129 lens."""

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

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256
from run_p0672_spent_rxj2129_absolute_raw_topology import (
    AbsoluteGridLens,
    PhysicalDeflectionGrid,
    exact_fit,
    global_topology,
    near_bound_count,
    topology_summary,
)
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_images

DEFAULT_CONFIG = ROOT / "configs" / "p0675_spent_rxj2129_compound_raw_topology.json"
MODELS = ("scalar_absolute_AQUAL", "compound_absolute_P0673")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    expected_status = "frozen_before_any_P0675_raw_lens_or_topology_score"
    if protocol.get("status") != expected_status:
        raise RuntimeError("P0675 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0674 parent no longer passes")
    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    field_path = ROOT / protocol["field_input"]
    with np.load(field_path) as data:
        axis_arcsec = data["axis_kpc"].astype(float) / scale
        grids = {
            "scalar_absolute_AQUAL": PhysicalDeflectionGrid(
                axis_arcsec,
                data["scalar_alpha_x_physical_arcsec"].astype(float),
                data["scalar_alpha_y_physical_arcsec"].astype(float),
            ),
            "compound_absolute_P0673": PhysicalDeflectionGrid(
                axis_arcsec,
                data["compound_alpha_x_physical_arcsec"].astype(float),
                data["compound_alpha_y_physical_arcsec"].astype(float),
            ),
        }
    lens = AbsoluteGridLens(raw, grids)
    fitted = {}
    predictions = []
    fit_rows = []
    parameter_rows = []
    for index, model in enumerate(MODELS):
        result = exact_fit(lens, model, training, heldout, protocol, index)
        fitted[model] = result
        predictions.append(
            pd.concat(
                [result["training_prediction"], result["heldout_prediction"]],
                ignore_index=True,
            )
        )
        fit_rows.append(
            {
                "model": model,
                "training_RMS_arcsec": result["training_score"][
                    "exact_radial_RMS_arcsec"
                ],
                "training_roots_converged": result["training_score"][
                    "converged_roots"
                ],
                "heldout_RMS_arcsec": result["heldout_score"][
                    "exact_radial_RMS_arcsec"
                ],
                "heldout_roots_converged": result["heldout_score"]["converged_roots"],
                "optimizer_cost": result["optimizer_cost"],
                "nuisance_parameters_near_bound": near_bound_count(result["parameters"]),
            }
        )
        parameter_rows.extend(
            {
                "model": model,
                "parameter": label,
                "value": float(value),
                "lower": float(lower),
                "upper": float(upper),
            }
            for label, value, lower, upper in zip(
                AbsoluteGridLens.labels,
                result["parameters"],
                AbsoluteGridLens.lower,
                AbsoluteGridLens.upper,
                strict=True,
            )
        )

    fit_scores = pd.DataFrame(fit_rows)
    topology_results = {}
    root_frames = []
    assignment_frames = []
    family_frames = []
    critical_maps = {}
    settings = protocol["global_topology"]
    for model in MODELS:
        roots, assignments, families, model_critical = global_topology(
            lens,
            model,
            fitted[model],
            images,
            settings,
        )
        root_frames.append(roots)
        assignment_frames.append(assignments)
        family_frames.append(families)
        critical_maps[model] = model_critical
        topology_results[model] = topology_summary(families)

    indexed = fit_scores.set_index("model")
    scalar_fit = indexed.loc["scalar_absolute_AQUAL"]
    compound_fit = indexed.loc["compound_absolute_P0673"]
    scalar_training = float(scalar_fit.training_RMS_arcsec)
    compound_training = float(compound_fit.training_RMS_arcsec)
    scalar_heldout = float(scalar_fit.heldout_RMS_arcsec)
    compound_heldout = float(compound_fit.heldout_RMS_arcsec)
    training_improvement = (
        1.0 - compound_training / scalar_training
        if np.isfinite(scalar_training) and np.isfinite(compound_training)
        else float("-inf")
    )
    heldout_worsening = (
        compound_heldout / scalar_heldout - 1.0
        if np.isfinite(scalar_heldout) and np.isfinite(compound_heldout)
        else float("inf")
    )
    comparator = read_json(ROOT / protocol["comparator_report"])
    compact_halo = float(
        comparator["model_scores"]["GR_plus_cluster_halo"]["heldout"][
            "exact_radial_RMS_arcsec"
        ]
    )
    compound_halo_ratio = (
        compound_heldout / compact_halo if np.isfinite(compound_heldout) else float("inf")
    )
    compound_topology = topology_results["compound_absolute_P0673"]
    gates = protocol["predeclared_progression_gates"]
    accounting = protocol["models"]
    gate_results = {
        "P0674_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0674_all_progression_gates_pass"]),
        "training_roots": int(compound_fit.training_roots_converged)
        == int(gates["compound_training_roots_converged"]),
        "heldout_roots": int(compound_fit.heldout_roots_converged)
        == int(gates["compound_heldout_roots_converged"]),
        "training_improvement": training_improvement
        >= float(gates["compound_training_RMS_improvement_fraction_vs_scalar_min"]),
        "heldout_stability": heldout_worsening
        <= float(gates["compound_heldout_RMS_worsening_fraction_vs_scalar_max"]),
        "heldout_absolute_RMS": compound_heldout
        <= float(gates["compound_heldout_RMS_arcsec_max"]),
        "compact_halo_comparison": compound_halo_ratio
        <= float(gates["compound_to_compact_halo_heldout_RMS_ratio_max"]),
        "no_missing_multiplicity": compound_topology["missing_multiplicity_families"]
        <= int(gates["compound_missing_multiplicity_families_max"]),
        "observable_surplus": compound_topology[
            "potentially_observable_surplus_families"
        ]
        <= int(gates["compound_potentially_observable_surplus_families_max"]),
        "acceptable_multiplicity": compound_topology[
            "exact_or_demagnified_only_families"
        ]
        >= int(gates["compound_exact_or_demagnified_only_families_min"]),
        "parity_diversity": compound_topology["parity_diverse_families"]
        >= int(gates["compound_parity_diverse_families_min"]),
        "critical_curves": compound_topology["critical_curve_present_families"]
        >= int(gates["compound_critical_curve_present_families_min"]),
        "nuisance_bounds": int(compound_fit.nuisance_parameters_near_bound)
        <= int(gates["compound_nuisance_parameters_near_bound_max"]),
        "no_fitted_gravity": int(accounting["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(
            accounting["photon_amplitudes_fit_to_RXJ2129"]
        )
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fit_scores.to_csv(output / "fit_scores.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "nuisance_parameters.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / "exact_predictions.csv",
        index=False,
    )
    roots = pd.concat(root_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    families = pd.concat(family_frames, ignore_index=True)
    roots.to_csv(output / "global_roots.csv", index=False)
    assignments.to_csv(output / "global_assignments.csv", index=False)
    families.to_csv(output / "family_topology.csv", index=False)

    report = {
        "report_version": "P0675-SPENT-RXJ2129-COMPOUND-RAW-TOPOLOGY-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "field_sha256": sha256(field_path),
        "coverage": {
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "source_families": int(images.source_family.nunique()),
            "ordinary_nuisance_parameters": 4,
            "profiled_source_coordinates": 14,
            "gravity_parameters": int(accounting["gravity_parameters_fit_to_RXJ2129"]),
            "photon_amplitudes": int(accounting["photon_amplitudes_fit_to_RXJ2129"]),
        },
        "fit_scores": fit_scores.to_dict(orient="records"),
        "comparisons": {
            "compound_training_improvement_fraction_vs_scalar": training_improvement,
            "compound_heldout_worsening_fraction_vs_scalar": heldout_worsening,
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "compound_to_compact_halo_heldout_RMS_ratio": compound_halo_ratio,
            "published_multi_halo_reference_RMS_arcsec": 0.29,
        },
        "topology": topology_results,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(fit_scores.model, fit_scores.training_RMS_arcsec, label="training")
    axes[0].scatter(
        fit_scores.model,
        fit_scores.heldout_RMS_arcsec,
        color="black",
        label="spent heldout",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="exact root RMS (arcsec)", title="Absolute raw-lens score")
    axes[0].legend()
    compound_families = families[
        families.variant_id.eq("compound_absolute_P0673")
    ]
    class_counts = compound_families.multiplicity_classification.value_counts()
    axes[1].bar(class_counts.index, class_counts.values)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="families", title="Compound multiplicity")
    determinant = critical_maps["compound_absolute_P0673"][1]
    half = float(settings["critical_grid_half_width_arcsec"])
    image = axes[2].imshow(
        np.sign(determinant).T,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[2].set(
        title="Family 1 Jacobian sign",
        xlabel="x (arcsec)",
        ylabel="y (arcsec)",
    )
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "p0675_compound_raw_topology.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0675 spent RX J2129 compound raw topology

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Scalar/compound training RMS: **{scalar_training:.4g} / {compound_training:.4g} arcsec**.
- Scalar/compound spent-heldout RMS: **{scalar_heldout:.4g} / {compound_heldout:.4g} arcsec**.
- Compound training improvement / heldout worsening: **{100 * training_improvement:+.3g}% / {100 * heldout_worsening:+.3g}%**.
- Compound missing / exact-or-demagnified / observable-surplus families: **{compound_topology['missing_multiplicity_families']} / {compound_topology['exact_or_demagnified_only_families']} / {compound_topology['potentially_observable_surplus_families']}**.
- Compound parity-diverse / critical-curve families: **{compound_topology['parity_diverse_families']} / {compound_topology['critical_curve_present_families']}** of 7.
- Compound/compact-halo heldout RMS ratio: **{compound_halo_ratio:.4g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
